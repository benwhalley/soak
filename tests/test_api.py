"""Tests for soak.api module."""

import tempfile
from pathlib import Path

import pytest

from soak import api
from soak.api import (CalibrateError, ShowError, get_pipeline, get_template,
                      list_pipelines, list_templates)


class TestListPipelines:
    """Tests for api.list_pipelines()."""

    def test_returns_list(self):
        """list_pipelines returns a list."""
        result = list_pipelines()
        assert isinstance(result, list)

    def test_contains_builtin_pipelines(self):
        """list_pipelines includes known built-in pipelines."""
        result = list_pipelines()
        # there should be at least one pipeline
        assert len(result) > 0
        # pipeline names should be strings
        assert all(isinstance(p, str) for p in result)

    def test_pipeline_names_have_no_extension(self):
        """Pipeline names should not include .soak extension."""
        result = list_pipelines()
        assert all(not p.endswith(".soak") for p in result)


class TestListTemplates:
    """Tests for api.list_templates()."""

    def test_returns_list(self):
        """list_templates returns a list."""
        result = list_templates()
        assert isinstance(result, list)

    def test_contains_builtin_templates(self):
        """list_templates includes known built-in templates."""
        result = list_templates()
        # there should be at least one template
        assert len(result) > 0
        # template names should be strings
        assert all(isinstance(t, str) for t in result)
        # should include common templates
        assert "simple" in result or any("simple" in t for t in result)

    def test_template_names_have_no_extension(self):
        """Template names should not include .html extension."""
        result = list_templates()
        assert all(not t.endswith(".html") for t in result)

    def test_excludes_partials(self):
        """Template list should exclude partials (files starting with _)."""
        result = list_templates()
        assert all(not t.startswith("_") for t in result)


class TestGetPipeline:
    """Tests for api.get_pipeline()."""

    def test_get_builtin_pipeline(self):
        """Can get content of a built-in pipeline."""
        pipelines = list_pipelines()
        if pipelines:
            content = get_pipeline(pipelines[0])
            assert isinstance(content, str)
            assert len(content) > 0

    def test_pipeline_not_found_raises_error(self):
        """ShowError raised when pipeline not found."""
        with pytest.raises(ShowError):
            get_pipeline("nonexistent_pipeline_xyz123")

    def test_get_pipeline_from_local_dir(self):
        """Can get pipeline from current working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # create a local pipeline
            pipeline_content = "name: test\nnodes: []"
            (tmppath / "local_test.soak").write_text(pipeline_content)

            result = get_pipeline("local_test", cwd=tmppath)
            assert result == pipeline_content

    def test_local_pipeline_takes_precedence(self):
        """Local pipeline takes precedence over built-in."""
        pipelines = list_pipelines()
        if not pipelines:
            pytest.skip("No built-in pipelines to test with")

        builtin_name = pipelines[0]
        local_content = "# local override"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / f"{builtin_name}.soak").write_text(local_content)

            result = get_pipeline(builtin_name, cwd=tmppath)
            assert result == local_content


class TestGetTemplate:
    """Tests for api.get_template()."""

    def test_get_builtin_template(self):
        """Can get content of a built-in template."""
        templates = list_templates()
        if templates:
            content = get_template(templates[0])
            assert isinstance(content, str)
            assert len(content) > 0

    def test_template_not_found_raises_error(self):
        """ShowError raised when template not found."""
        with pytest.raises(ShowError):
            get_template("nonexistent_template_xyz123")

    def test_get_template_from_local_dir(self):
        """Can get template from current working directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            template_content = "<html>test</html>"
            (tmppath / "local_test.html").write_text(template_content)

            result = get_template("local_test", cwd=tmppath)
            assert result == template_content


class TestCalibrateValidation:
    """Tests for api.calibrate() input validation."""

    def test_requires_input_or_paraphrases(self):
        """calibrate raises error when neither input nor paraphrases provided."""
        with pytest.raises(CalibrateError, match="Must provide either"):
            api.calibrate()

    def test_requires_prompt_with_input(self):
        """calibrate raises error when input_csv provided without prompt."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1\nvalue1\n")
            f.flush()
            with pytest.raises(CalibrateError, match="prompt required"):
                api.calibrate(input_csv=Path(f.name))

    def test_mutually_exclusive_input_and_paraphrases(self):
        """calibrate raises error when both input_csv and paraphrases_csv provided."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f1:
            f1.write(b"col1\nvalue1\n")
            f1.flush()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f2:
                f2.write(b"col1\nvalue1\n")
                f2.flush()
                with tempfile.NamedTemporaryFile(suffix=".sd", delete=False) as prompt:
                    prompt.write(b"test prompt")
                    prompt.flush()
                    with pytest.raises(CalibrateError, match="Cannot use both"):
                        api.calibrate(
                            input_csv=Path(f1.name),
                            paraphrases_csv=Path(f2.name),
                            prompt=Path(prompt.name),
                        )

    def test_mutually_exclusive_head_and_sample(self):
        """calibrate raises error when both head and sample provided."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"original,text,category\na,b,paraphrase\n")
            f.flush()
            with pytest.raises(CalibrateError, match="Cannot use both head and sample"):
                api.calibrate(paraphrases_csv=Path(f.name), head=10, sample=5)

    def test_invalid_method_raises_error(self):
        """calibrate raises error for invalid method."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"original,text,category\na,b,paraphrase\n")
            f.flush()
            with pytest.raises(CalibrateError, match="Invalid method"):
                api.calibrate(paraphrases_csv=Path(f.name), method="invalid")


class TestAPIExports:
    """Tests that all expected functions are exported from api module."""

    def test_run_exported(self):
        """api.run is exported."""
        assert hasattr(api, "run")
        assert callable(api.run)

    def test_run_async_exported(self):
        """api.run_async is exported."""
        assert hasattr(api, "run_async")
        assert callable(api.run_async)

    def test_compare_exported(self):
        """api.compare is exported."""
        assert hasattr(api, "compare")
        assert callable(api.compare)

    def test_compare_strings_exported(self):
        """api.compare_strings is exported."""
        assert hasattr(api, "compare_strings")
        assert callable(api.compare_strings)

    def test_render_exported(self):
        """api.render is exported."""
        assert hasattr(api, "render")
        assert callable(api.render)

    def test_load_exported(self):
        """api.load is exported."""
        assert hasattr(api, "load")
        assert callable(api.load)

    def test_export_functions_exported(self):
        """api.export_pdf and api.export_xlsx are exported."""
        assert hasattr(api, "export_pdf")
        assert hasattr(api, "export_xlsx")

    def test_coverage_exported(self):
        """api.coverage is exported."""
        assert hasattr(api, "coverage")
        assert callable(api.coverage)

    def test_credentials_functions_exported(self):
        """Credential management functions are exported."""
        assert hasattr(api, "set_credentials")
        assert hasattr(api, "get_credentials")
        assert hasattr(api, "clear_credentials")
        assert hasattr(api, "credentials")

    def test_show_functions_exported(self):
        """Show functions are exported."""
        assert hasattr(api, "list_pipelines")
        assert hasattr(api, "list_templates")
        assert hasattr(api, "get_pipeline")
        assert hasattr(api, "get_template")

    def test_calibrate_exported(self):
        """api.calibrate is exported."""
        assert hasattr(api, "calibrate")
        assert callable(api.calibrate)

    def test_error_classes_exported(self):
        """Error classes are exported."""
        assert hasattr(api, "RunError")
        assert hasattr(api, "CompareError")
        assert hasattr(api, "RenderError")
        assert hasattr(api, "ExportError")
        assert hasattr(api, "CoverageError")
        assert hasattr(api, "CredentialsError")
        assert hasattr(api, "ShowError")
        assert hasattr(api, "CalibrateError")

    def test_result_classes_exported(self):
        """Result classes are exported."""
        assert hasattr(api, "RunResult")
        assert hasattr(api, "CompareResult")
        assert hasattr(api, "CoverageResult")
        assert hasattr(api, "CostSummary")
        assert hasattr(api, "CalibrationResult")
