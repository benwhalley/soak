import logging
import re
from pathlib import Path
from typing import Union

import yaml

from soak.models import DAGConfig, DAGNode, QualitativeAnalysisPipeline
# Import all node types to rebuild their schemas with BatchList
from soak.models.nodes import (Batch, Classifier, Cluster, Filter, GroupBy, Map, Reduce,
                               Scrub, Split, Transform, Ungroup, VerifyQuotes)
# Ensure BatchList is imported before model_rebuild to resolve forward references
from soak.models.nodes.batch import BatchList  # noqa: F401

logger = logging.getLogger(__name__)

# Rebuild all node types to resolve "BatchList" forward reference in OutputUnion
for node_class in [
    DAGConfig,
    DAGNode,
    Batch,
    Classifier,
    Cluster,
    Filter,
    GroupBy,
    Map,
    Reduce,
    Scrub,
    Split,
    Transform,
    Ungroup,
    VerifyQuotes,
]:
    node_class.model_rebuild(force=True)


def extract_templates_(text):
    pattern = re.compile(
        r"---#(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\n(?P<content>.*?)(?=^---#|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    sections = {m["name"]: m["content"].strip() for m in pattern.finditer(text)}
    return sections


# extract_templates_(open("new.soak").read())


def load_template_bundle(template: Union[Path, str]) -> QualitativeAnalysisPipeline:
    """Load pipeline from .soak file with hybrid inline/external template support.

    Template resolution priority:
    1. Inline templates (---#node_name sections)
    2. External .sd files in search paths:
       - CWD where .soak file is
       - templates/ subdirectory in CWD
       - Directories specified in template_dirs YAML field
       - soak/templates/ (package defaults)
    """
    from soak.template_resolution import TemplateResolver

    if isinstance(template, str):
        text = template
        template_path = Path.cwd()  # fallback for string input
    else:
        text = template.read_text()
        template_path = Path(template)

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

    # Create template resolver
    extra_dirs = loaded.get("template_dirs", [])
    resolver = TemplateResolver(template_path, extra_dirs=extra_dirs)

    # Inject templates into node dicts before validation
    if "nodes" in loaded:
        for node in loaded["nodes"]:
            node_name = node.get("name")

            # Priority 1: Inline template (---#node_name)
            if node_name and node_name in templates:
                node["template"] = templates[node_name]
                continue

            # Priority 2: Explicit template field (template: filename.sd)
            if "template" in node and node["template"]:
                template_file = node["template"]
                try:
                    node["template"] = resolver.load(template_file)
                    logger.debug(
                        f"Loaded explicit template '{template_file}' for node '{node_name}'"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed to load template '{template_file}' for node '{node_name}': {e}"
                    )
                    raise

            # Priority 3: Convention -- look for {node_name}.sd
            external_template = resolver.try_load(f"{node_name}.sd")
            if external_template:
                node["template"] = external_template
                logger.debug(
                    f"Loaded external template '{node_name}.sd' for node '{node_name}'"
                )

    pipeline = QualitativeAnalysisPipeline.model_validate(loaded)
    for i in pipeline.nodes:
        i.validate_template()
        i.dag = pipeline

    # compute and set the pipeline version (version + content hash)
    pipeline.set_pipeline_version()
    logger.debug(f"Pipeline version: {pipeline.pipeline_version}")

    # check if pipeline includes a Scrub node and warn if not (unless scrub: false is set)
    has_scrub_node = any(node.type == "Scrub" for node in pipeline.nodes)
    if not has_scrub_node and pipeline.scrub is not False:
        logger.warning(
            "\n⚠️  Pipeline does not include a Scrub node for PII detection/redaction.\n"
            "   Consider adding a Scrub node to protect sensitive information:\n"
            "   \n"
            "   nodes:\n"
            "     - name: scrubbed_docs\n"
            "       type: Scrub\n"
            "       inputs: [documents]\n"
            "       redact: true\n"
            "   \n"
            "   To suppress this warning, add 'scrub: false' to your pipeline YAML.\n"
        )

    return pipeline


def pipeline_to_template_bundle(pipeline: QualitativeAnalysisPipeline) -> str:
    """
    Convert a QualitativeAnalysisPipeline back to template bundle format.

    Returns:
        str: template bundle content in YAML + template blocks format
    """

    # use model_dump to get the pipeline data, excluding computed/internal fields

    dumped = pipeline.model_dump(
        exclude={"nodes": {"__all__": {"template"}}, "config": True}
    )
    templates = {k.name: k.template for k in pipeline.nodes if k.template}

    # generate yaml content
    yaml_content = yaml.dump(dumped, default_flow_style=False, sort_keys=False)

    # add template blocks

    template_blocks = "\n".join([f"---#{k}\n\n{v}\n\n" for k, v in templates.items()])

    return yaml_content + "\n\n\n" + template_blocks
