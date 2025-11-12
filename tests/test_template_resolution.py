"""Tests for hybrid inline/external template resolution."""

import tempfile
from pathlib import Path

import pytest

from soak.specs import load_template_bundle
from soak.template_resolution import TemplateNotFoundError, TemplateResolver


class TestTemplateResolver:
    """Test TemplateResolver search path logic."""

    def test_basic_resolution(self, tmp_path):
        """Test finding template in CWD."""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text("name: test")

        template_file = tmp_path / "summary.sd"
        template_file.write_text("Summarize this: {{input}}")

        resolver = TemplateResolver(pipeline_path)
        content = resolver.load("summary.sd")
        assert content == "Summarize this: {{input}}"

    def test_templates_subdir(self, tmp_path):
        """Test finding template in templates/ subdirectory."""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text("name: test")

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        template_file = templates_dir / "summary.sd"
        template_file.write_text("Template content")

        resolver = TemplateResolver(pipeline_path)
        content = resolver.load("summary.sd")
        assert content == "Template content"

    def test_extra_dirs(self, tmp_path):
        """Test custom template_dirs."""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text("name: test")

        custom_dir = tmp_path / "custom_templates"
        custom_dir.mkdir()
        template_file = custom_dir / "summary.sd"
        template_file.write_text("Custom template")

        resolver = TemplateResolver(pipeline_path, extra_dirs=[str(custom_dir)])
        content = resolver.load("summary.sd")
        assert content == "Custom template"

    def test_not_found_error(self, tmp_path):
        """Test helpful error message when template not found."""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text("name: test")

        resolver = TemplateResolver(pipeline_path)
        with pytest.raises(TemplateNotFoundError) as exc_info:
            resolver.load("missing.sd")

        assert "missing.sd" in str(exc_info.value)
        assert "Searched in:" in str(exc_info.value)

    def test_priority_order(self, tmp_path):
        """Test that CWD takes priority over templates/ subdirectory."""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text("name: test")

        # Create template in both CWD and templates/
        (tmp_path / "test.sd").write_text("CWD version")
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        (templates_dir / "test.sd").write_text("templates/ version")

        resolver = TemplateResolver(pipeline_path)
        content = resolver.load("test.sd")
        assert content == "CWD version"  # CWD should win


class TestHybridTemplates:
    """Test hybrid inline/external template loading in pipelines."""

    def test_inline_template_still_works(self, tmp_path):
        """Ensure backward compatibility -- inline templates still work."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    inputs: []

---#summarize
Summarize: {{input}}
[[summary]]
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template.strip() == "Summarize: {{input}}\n[[summary]]"

    def test_external_template_convention(self, tmp_path):
        """Test loading external template by convention ({node_name}.sd)."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    inputs: []
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        # Create external template
        template_file = tmp_path / "summarize.sd"
        template_file.write_text("External template: {{input}}")

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template == "External template: {{input}}"

    def test_explicit_template_field(self, tmp_path):
        """Test explicit template: field in YAML."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    template: custom.sd
    inputs: []
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        # Create custom template
        template_file = tmp_path / "custom.sd"
        template_file.write_text("Custom template content")

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template == "Custom template content"

    def test_templates_subdir(self, tmp_path):
        """Test loading from templates/ subdirectory."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    inputs: []
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        # Create template in templates/ subdir
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        template_file = templates_dir / "summarize.sd"
        template_file.write_text("Template from subdir")

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template == "Template from subdir"

    def test_custom_template_dirs(self, tmp_path):
        """Test template_dirs YAML field."""
        custom_dir = tmp_path / "my_templates"
        custom_dir.mkdir()
        template_file = custom_dir / "summarize.sd"
        template_file.write_text("Custom dir template")

        pipeline_content = f"""name: test_pipeline
template_dirs:
  - {custom_dir}
nodes:
  - name: summarize
    type: Map
    inputs: []
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template == "Custom dir template"

    def test_inline_takes_priority(self, tmp_path):
        """Test that inline template takes priority over external file."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    inputs: []

---#summarize
Inline template wins
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        # Also create external file (should be ignored)
        template_file = tmp_path / "summarize.sd"
        template_file.write_text("External template loses")

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert "Inline template wins" in pipeline.nodes[0].template

    def test_explicit_overrides_convention(self, tmp_path):
        """Test explicit template: field overrides convention."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: summarize
    type: Map
    template: other.sd
    inputs: []
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        # Create both templates
        (tmp_path / "summarize.sd").write_text("Convention template")
        (tmp_path / "other.sd").write_text("Explicit template")

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template == "Explicit template"

    def test_nodes_without_templates_still_work(self, tmp_path):
        """Test that nodes not requiring templates (Split, Reduce) work fine."""
        pipeline_content = """name: test_pipeline
nodes:
  - name: chunks
    type: Split
    chunk_size: 100
"""
        pipeline_path = tmp_path / "test.soak"
        pipeline_path.write_text(pipeline_content)

        pipeline = load_template_bundle(pipeline_path)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].template is None
