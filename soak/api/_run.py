"""Run pipeline implementation for soak API."""

from __future__ import annotations

import asyncio
import json
import shutil
from glob import glob
from pathlib import Path
from typing import Optional, Union

from struckdown import LLMCredentials

from ..cli._common import PIPELINE_DIR
from ..document_utils import unpack_zip_to_temp_paths_if_needed
from ..helpers import derive_input_source, hash_run_config, resolve_pipeline
from ..specs import load_template_bundle
from ._credentials import CredentialsError, get_credentials
from ._results import RunResult


class RunError(Exception):
    """Error during pipeline execution."""

    pass


# package data directory
SOAK_DATA_DIR = Path(__file__).parent.parent / "soak-data"


def _strip_soak_data_prefix(path_str: str) -> str:
    """Strip 'soak-data/' prefix if present."""
    if path_str.startswith("soak-data/"):
        return path_str[len("soak-data/") :]
    return path_str


def _expand_inputs(
    inputs: Union[str, list[Union[str, Path]]], cwd: Optional[Path] = None
) -> list[Path]:
    """Expand glob patterns and convert to Path objects.

    Searches in order:
    1. Current working directory (or cwd parameter)
    2. Package's soak-data directory (for bundled sample data)
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    cwd = cwd or Path.cwd()

    paths = []
    for inp in inputs:
        inp_str = str(inp)
        if "*" in inp_str or "?" in inp_str:
            # try glob relative to cwd first
            if not Path(inp_str).is_absolute():
                pattern = str(cwd / inp_str)
            else:
                pattern = inp_str
            expanded = glob(pattern)

            # if nothing found, try package soak-data directory
            if not expanded and not Path(inp_str).is_absolute():
                # strip soak-data/ prefix if present to avoid soak-data/soak-data/
                stripped = _strip_soak_data_prefix(inp_str)
                package_pattern = str(SOAK_DATA_DIR / stripped)
                expanded = glob(package_pattern)

            paths.extend(Path(p) for p in expanded)
        else:
            # resolve path relative to cwd
            inp_path = Path(inp_str)
            if not inp_path.is_absolute():
                # try cwd first
                cwd_path = cwd / inp_path
                if cwd_path.exists():
                    paths.append(cwd_path)
                else:
                    # try package soak-data (strip prefix if present)
                    stripped = _strip_soak_data_prefix(inp_str)
                    package_path = SOAK_DATA_DIR / stripped
                    if package_path.exists():
                        paths.append(package_path)
                    else:
                        # add anyway, will fail later with clear error
                        paths.append(cwd_path)
            else:
                paths.append(inp_path)

    return paths


async def run_async(
    pipeline: str,
    inputs: Union[str, list[Union[str, Path]]],
    *,
    context: Optional[dict[str, str]] = None,
    output: Optional[Union[str, Path]] = None,
    model: Optional[Union[str, dict[str, str]]] = None,
    seed: Optional[int] = None,
    sample: Optional[int] = None,
    head: Optional[int] = None,
    skip_nodes: Optional[list[str]] = None,
    stop_at: Optional[str] = None,
    embedding_model: Optional[str] = None,
    timeout: int = 90,
    progress: Optional[bool] = None,
    force: bool = False,
    include_documents: bool = False,
    instructor_mode: str = "json",
    cwd: Optional[Path] = None,
) -> RunResult:
    """Run a pipeline on input files (async version).

    Args:
        pipeline: Pipeline name (e.g., "zs") or path to .soak file
        inputs: Glob pattern or list of file paths
        context: Context variables to override (key=value pairs)
        output: Output folder name (default: derived from pipeline)
        model: Model name or dict of aliases {"default": "gpt-4", "best": "gpt-5"}
        seed: Random seed for reproducibility
        sample: Randomly sample N documents
        head: Take first N documents
        skip_nodes: Nodes to skip during execution
        stop_at: Stop before this node runs
        embedding_model: Override embedding model
        timeout: LLM timeout in seconds
        progress: Show progress bars (auto-detected if None)
        force: Overwrite existing outputs
        include_documents: Include source docs in output
        instructor_mode: Instructor mode for structured outputs
        cwd: Working directory for resolving paths

    Returns:
        RunResult with .analysis, .themes, .codes, .to_html(), .to_json()

    Raises:
        RunError: If pipeline execution fails
        CredentialsError: If credentials are missing
    """
    cwd = cwd or Path.cwd()

    # validate mutually exclusive options
    if sample is not None and head is not None:
        raise RunError("sample and head are mutually exclusive")

    # get credentials
    creds = get_credentials(cwd)

    # resolve and load pipeline
    pipyml = resolve_pipeline(pipeline, cwd, PIPELINE_DIR)
    pipe = load_template_bundle(pipyml)

    # determine output folder
    if output is None:
        output_name = Path(pipyml).stem
    else:
        output_name = str(output)

    dump_path = Path(f"{output_name}_dump")

    # check for existing output
    if dump_path.exists() and not force:
        raise RunError(
            f"Output folder already exists: {dump_path}/. Use force=True to overwrite."
        )
    elif dump_path.exists() and force:
        shutil.rmtree(dump_path)

    # set context variables
    if context:
        for key, value in context.items():
            pipe.default_context[key] = value

    # configure model
    model_aliases = {}
    if model:
        if isinstance(model, str):
            model_aliases["default"] = model
            pipe.config.model_name = model
        else:
            model_aliases = model
            if "default" in model:
                pipe.config.model_name = model["default"]
        pipe.config.models = {**pipe.config.models, **model_aliases}

    # configure other options
    if seed is not None:
        pipe.config.seed = seed
    if embedding_model is not None:
        pipe.config.embedding_model = embedding_model
    if sample is not None:
        pipe.config.sample_n = sample
    if head is not None:
        pipe.config.head_n = head
    if skip_nodes:
        pipe.config.skip_nodes = skip_nodes
    if stop_at:
        pipe.config.stop_at_node = stop_at

    pipe.config.llm_timeout = timeout
    pipe.config.show_progress = progress if progress is not None else False
    pipe.config.llm_credentials = LLMCredentials(
        api_key=creds.api_key,
        base_url=creds.base_url,
        instructor_mode=instructor_mode,
    )

    # enable incremental export
    pipe.config.export_enabled = True
    pipe.config.export_folder = dump_path

    # build metadata
    input_paths = _expand_inputs(inputs, cwd)
    config_hash = hash_run_config(
        input_files=input_paths,
        model_name=model_aliases.get("default") if model_aliases else None,
        context=list(f"{k}={v}" for k, v in (context or {}).items()),
        template=[],
    )
    pipe.config.export_metadata = {
        "pipeline_file": str(pipyml),
        "pipeline_version": pipe.pipeline_version,
        "model_aliases": model_aliases or {},
        "unique_id": config_hash,
        "sample_n": sample,
        "head_n": head,
        "seed": seed,
        "embedding_model": embedding_model,
    }
    if context:
        pipe.config.export_metadata["context_overrides"] = context

    # load documents
    with unpack_zip_to_temp_paths_if_needed(input_paths) as docfiles:
        if not docfiles:
            raise RunError(f"No files found matching input patterns: {inputs}")

        pipe.config.document_paths = docfiles
        pipe.config.input_source = derive_input_source(docfiles)
        pipe.config.documents = pipe.config.load_documents()

    # run pipeline
    analysis, errors = await pipe.run()

    # remove documents from output if not requested
    if not include_documents:
        analysis.config.documents = []

    # get cost summary
    cost_summary = analysis.get_cost_summary()

    # write final JSON output
    json_path = dump_path / f"{output_name}.json"
    json_path.write_text(json.dumps(analysis.get_model_dump(), indent=2))

    return RunResult(
        pipeline=analysis,
        output_folder=dump_path,
        errors=errors or [],
        _cost_summary=cost_summary,
    )


def run(
    pipeline: str,
    inputs: Union[str, list[Union[str, Path]]],
    *,
    context: Optional[dict[str, str]] = None,
    output: Optional[Union[str, Path]] = None,
    model: Optional[Union[str, dict[str, str]]] = None,
    seed: Optional[int] = None,
    sample: Optional[int] = None,
    head: Optional[int] = None,
    skip_nodes: Optional[list[str]] = None,
    stop_at: Optional[str] = None,
    embedding_model: Optional[str] = None,
    timeout: int = 90,
    progress: Optional[bool] = None,
    force: bool = False,
    include_documents: bool = False,
    instructor_mode: str = "json",
    cwd: Optional[Path] = None,
) -> RunResult:
    """Run a pipeline on input files.

    Args:
        pipeline: Pipeline name (e.g., "zs") or path to .soak file
        inputs: Glob pattern or list of file paths
        context: Context variables to override (key=value pairs)
        output: Output folder name (default: derived from pipeline)
        model: Model name or dict of aliases {"default": "gpt-4", "best": "gpt-5"}
        seed: Random seed for reproducibility
        sample: Randomly sample N documents
        head: Take first N documents
        skip_nodes: Nodes to skip during execution
        stop_at: Stop before this node runs
        embedding_model: Override embedding model
        timeout: LLM timeout in seconds
        progress: Show progress bars (auto-detected if None)
        force: Overwrite existing outputs
        include_documents: Include source docs in output
        instructor_mode: Instructor mode for structured outputs
        cwd: Working directory for resolving paths

    Returns:
        RunResult with .analysis, .themes, .codes, .to_html(), .to_json()

    Raises:
        RunError: If pipeline execution fails
        CredentialsError: If credentials are missing
    """
    return asyncio.run(
        run_async(
            pipeline,
            inputs,
            context=context,
            output=output,
            model=model,
            seed=seed,
            sample=sample,
            head=head,
            skip_nodes=skip_nodes,
            stop_at=stop_at,
            embedding_model=embedding_model,
            timeout=timeout,
            progress=progress,
            force=force,
            include_documents=include_documents,
            instructor_mode=instructor_mode,
            cwd=cwd,
        )
    )
