"""Command-line interface for running qualitative analysis pipelines."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from struckdown import LLMCredentials

from .document_utils import unpack_zip_to_temp_paths_if_needed
from .specs import load_template_bundle

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("struckdown").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger().setLevel(logging.INFO)


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).parent / "pipelines"


app = typer.Typer()


def resolve_pipeline(pipeline: str, localdir: Path, pipelinedir: Path) -> Path:
    candidates = [
        localdir / pipeline,
        localdir / f"{pipeline}.yaml",
        localdir / f"{pipeline}.yml",
        pipelinedir / f"{pipeline}",
        pipelinedir / f"{pipeline}.yaml",
        pipelinedir / f"{pipeline}.yml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Pipeline file not found. Tried: {candidates}")

@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Pipeline name to run (e.g., 'poc')"),
    input_files: list[str] = typer.Argument(
        ..., help="File patterns or zip files (supports globs like '*.txt')"
    ),
    model_name: str = typer.Option("litellm/gpt-4.1-mini", help="LLM model name"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path name (without extensions) (stdout if not specified)",
    ),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or html"),
    include_documents: bool = typer.Option(
        False, "--include-documents", help="Include original documents in output"
    ),
):
    """Run a pipeline on input files."""

    pipyml = resolve_pipeline(pipeline, Path.cwd(), PIPELINE_DIR)        
    print(f"Loading pipeline from {pipyml}", file=sys.stderr)
    pipeline = load_template_bundle(pipyml)

    pipeline.config.model_name = model_name
    pipeline.config.llm_credentials = LLMCredentials()

    with unpack_zip_to_temp_paths_if_needed(input_files) as docfiles:
        pipeline.config.document_paths = docfiles
        pipeline.config.documents = pipeline.config.load_documents()
    
    try:
        analysis, errors = asyncio.run(pipeline.run())
        if errors:
            logger.error(f"Errors during pipeline execution: {errors}")
            logger.warning("Entering pdb for debugging")

    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)

    # analysis.config.llm_credentials.api_key = (
    #     analysis.config.llm_credentials.api_key[:5] + "***"
    # )

    # remove documents from output if not requested
    if not include_documents:
        analysis.config.documents = []
    
    
    # generate output content based on format
    # import pdb; pdb.set_trace()
    jsoncontent = analysis.model_dump_json()
    htmlcontent = analysis.to_html()

    # output to stdout or file
    if output is None:
        if format == "json":
            print(jsoncontent)
        elif format == "html":
            print(htmlcontent)
        else:
            raise typer.BadParameter("Format must be 'json' or 'html' or specify output file name")

    else:
        print(f"Writing output to {output}.json and {output}.html")
        with open(output + ".html", "w", encoding="utf-8") as f:
            f.write(htmlcontent)
        with open(output + ".json", "w", encoding="utf-8") as f:
            f.write(jsoncontent)



if __name__ == "__main__":
    app()
