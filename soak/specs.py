import logging
import re
from pathlib import Path
from typing import Union

import yaml

from soak.models import DAGNode, QualitativeAnalysisPipeline

logger = logging.getLogger(__name__)

DAGNode.model_rebuild(force=True)


def extract_templates_(text):
    pattern = re.compile(
        r"---#(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\n(?P<content>.*?)(?=^---#|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    sections = {m["name"]: m["content"].strip() for m in pattern.finditer(text)}
    return sections


# extract_templates_(open("new.yaml").read())


def load_template_bundle(template: Union[Path, str]) -> QualitativeAnalysisPipeline:
    if isinstance(template, str):
        text = template
    else:
        text = template.read_text()

    # try to parse YAML first
    BLOCK_DELIM_RE = re.compile(r"^---#(\w+)", re.MULTILINE)

    match = BLOCK_DELIM_RE.search(text)
    if match:
        # try to parse as template bundle
        yaml_text = text[: match.start()]
        templates = extract_templates_(text)
    else:
        # parse only yaml
        yaml_text = text
        templates = {}

    loaded = yaml.safe_load(yaml_text)
    pipeline = QualitativeAnalysisPipeline.model_validate(loaded)
    for i in pipeline.nodes:
        i.template_text = templates.get(i.name, i.template_text)
        i.validate_template()
        i.dag = pipeline

    return pipeline


def pipeline_to_template_bundle(pipeline: QualitativeAnalysisPipeline) -> str:
    """
    Convert a QualitativeAnalysisPipeline back to template bundle format.

    Returns:
        str: template bundle content in YAML + template blocks format
    """

    # use model_dump to get the pipeline data, excluding computed/internal fields

    dumped = pipeline.model_dump(
        exclude={"nodes": {"__all__": {"template_text"}}, "config": True}
    )
    templates = {k.name: k.template_text for k in pipeline.nodes if k.template_text}

    # generate yaml content
    yaml_content = yaml.dump(dumped, default_flow_style=False, sort_keys=False)

    # add template blocks

    template_blocks = "\n".join([f"---#{k}\n\n{v}\n\n" for k, v in templates.items()])

    return yaml_content + "\n\n\n" + template_blocks
