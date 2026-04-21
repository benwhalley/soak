"""Integration test: run full thematic analysis pipeline with 4 documents.

This tests the complete data flow: template rendering, StruckdownResult handling,
_TemplateProxy wrapping, and node output serialisation.

Requires LLM_API_KEY and LLM_API_BASE environment variables to be set.
"""

import os
from pathlib import Path

import pytest


@pytest.mark.anyio
@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set -- skipping live LLM test",
)
async def test_zs_pipeline_4_docs():
    """Run the full zs.soak thematic analysis pipeline with 4 airline review documents."""
    from struckdown import LLMCredentials

    from soak.specs import load_template_bundle

    # Load pipeline
    pipeline_path = (
        Path(__file__).parent.parent
        / "soak"
        / "pipelines"
        / "thematic_analysis"
        / "zs.soak"
    )
    assert pipeline_path.exists(), f"Pipeline not found: {pipeline_path}"
    dag = load_template_bundle(pipeline_path)

    # Load 4 example documents
    data_dir = (
        Path(__file__).parent.parent
        / "examples"
        / "airline-reviews"
        / "data"
        / "economy"
    )
    doc_files = sorted(data_dir.glob("*.txt"))[:4]
    assert len(doc_files) == 4, f"Expected 4 docs, found {len(doc_files)}"

    dag.config.document_paths = [str(f) for f in doc_files]
    dag.config.llm_credentials = LLMCredentials()
    dag.config.show_progress = False

    # Run the pipeline
    result, error = await dag.run()

    # Check no errors
    assert error is None, f"Pipeline failed: {error}"

    # Check all nodes completed
    for node in dag.nodes:
        assert node.output is not None, f"Node {node.name} has no output"

    # Check themes node output is a list containing a StruckdownResult
    from struckdown import StruckdownResult

    themes_node = dag.nodes_dict["themes"]
    assert isinstance(themes_node.output, list)
    assert len(themes_node.output) == 1
    cr = themes_node.output[0]
    assert isinstance(cr, StruckdownResult)
    assert "themes" in cr.results

    # Check the themes slot has Theme objects
    from soak.models.base import Theme

    themes_output = cr.results["themes"].output
    assert isinstance(themes_output, list)
    assert len(themes_output) >= 1
    assert isinstance(themes_output[0], Theme)

    # Check narrative node completed with text
    narrative_node = dag.nodes_dict["narrative"]
    assert isinstance(narrative_node.output, list)
    narrative_cr = narrative_node.output[0]
    assert isinstance(narrative_cr, StruckdownResult)
    assert "narrative" in narrative_cr.results
    narrative_text = narrative_cr.results["narrative"].output
    assert isinstance(narrative_text, str)
    assert len(narrative_text) > 100

    # Check _prepare_for_template works correctly
    from soak.models.nodes.base import _prepare_for_template

    # Transform output -> single proxy
    prepared_themes = _prepare_for_template(
        themes_node.output, source_node_type="Transform"
    )
    from soak.models.nodes.base import _TemplateProxy

    assert isinstance(prepared_themes, _TemplateProxy)

    # Proxy attribute access wraps [Theme] into Themes
    from soak.models.base import Themes

    themes_obj = prepared_themes.themes
    assert isinstance(themes_obj, Themes)
    assert len(themes_obj) >= 1

    # Map output -> list of proxies
    theme_groups_node = dag.nodes_dict["theme_groups"]
    prepared_groups = _prepare_for_template(
        theme_groups_node.output, source_node_type="Map"
    )
    assert isinstance(prepared_groups, list)
    assert all(isinstance(p, _TemplateProxy) for p in prepared_groups)

    # Each proxy's .themes returns Themes
    for proxy in prepared_groups:
        batch_themes = proxy.themes
        assert isinstance(batch_themes, Themes)
        assert len(batch_themes) >= 1

    print(f"Pipeline completed successfully:")
    print(f"  Themes: {len(themes_output)}")
    print(f"  Narrative: {len(narrative_text)} chars")
    print(f"  Theme groups: {len(prepared_groups)} batches")
