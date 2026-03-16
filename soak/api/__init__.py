"""Programmatic API for soak.

This module provides a clean Python interface for running soak pipelines
and other operations without going through the CLI.

Basic usage:

    from soak import api

    # Run a pipeline
    result = api.run("zs", "data/*.txt")
    print(f"Found {len(result.themes)} themes")

    # With options
    result = api.run(
        "zs",
        "interviews/*.txt",
        context={"persona": "...", "research_question": "..."},
        output="my_analysis",
        seed=42,
    )

    # Access results
    for theme in result.themes:
        print(f"- {theme.name}")

    # Export
    html = result.to_html("simple")
    result.save("output.json")

Credentials are resolved automatically from:
1. api.set_credentials() or api.credentials() context manager
2. Environment variables (LLM_API_KEY, LLM_API_BASE)
3. .env file in current directory

You can set credentials explicitly:

    api.set_credentials(api_key="...", base_url="...")

    # Or use a context manager
    with api.credentials(api_key="..."):
        result = api.run(...)
"""

from ._calibrate import CalibrateError, CalibrationResult, calibrate
from ._compare import CompareError, compare, compare_strings
from ._coverage import CoverageError, coverage
from ._credentials import (Credentials, CredentialsError, clear_credentials,
                           credentials, get_credentials, set_credentials)
from ._export import ExportError, export_pdf, export_xlsx
from ._render import RenderError, load, render
from ._results import CompareResult, CostSummary, CoverageResult, RunResult
from ._run import RunError, run, run_async
from ._show import (ShowError, get_pipeline, get_template, list_pipelines,
                    list_templates)

__all__ = [
    # run
    "run",
    "run_async",
    "RunResult",
    "RunError",
    # compare
    "compare",
    "compare_strings",
    "CompareResult",
    "CompareError",
    # render
    "render",
    "load",
    "RenderError",
    # export
    "export_pdf",
    "export_xlsx",
    "ExportError",
    # coverage
    "coverage",
    "CoverageResult",
    "CoverageError",
    # credentials
    "set_credentials",
    "clear_credentials",
    "credentials",
    "get_credentials",
    "Credentials",
    "CredentialsError",
    # results
    "CostSummary",
    # show
    "list_pipelines",
    "list_templates",
    "get_pipeline",
    "get_template",
    "ShowError",
    # calibrate
    "calibrate",
    "CalibrationResult",
    "CalibrateError",
]
