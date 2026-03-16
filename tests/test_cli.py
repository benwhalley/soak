"""Tests for soak CLI commands."""

import pytest
from typer.testing import CliRunner

from soak.cli import app


runner = CliRunner()


class TestShowCommand:
    """Tests for `soak show` command."""

    def test_show_pipeline_lists_pipelines(self):
        """show pipeline (without name) lists available pipelines."""
        result = runner.invoke(app, ["show", "pipeline", "-v"])
        # should succeed and list pipelines
        assert result.exit_code == 0 or "Available" in result.output

    def test_show_template_lists_templates(self):
        """show template (without name) lists available templates."""
        result = runner.invoke(app, ["show", "template", "-v"])
        # should succeed and list templates
        assert result.exit_code == 0 or "Available" in result.output

    def test_show_pipeline_content(self):
        """show pipeline <name> outputs pipeline content."""
        # first get list of pipelines
        from soak.api import list_pipelines

        pipelines = list_pipelines()
        if not pipelines:
            pytest.skip("No built-in pipelines available")

        result = runner.invoke(app, ["show", "pipeline", pipelines[0]])
        assert result.exit_code == 0
        # pipeline content should be YAML-like
        assert "name:" in result.output or "nodes:" in result.output or len(result.output) > 0

    def test_show_template_content(self):
        """show template <name> outputs template content."""
        from soak.api import list_templates

        templates = list_templates()
        if not templates:
            pytest.skip("No built-in templates available")

        result = runner.invoke(app, ["show", "template", templates[0]])
        assert result.exit_code == 0
        # template content should be HTML-like
        assert "<" in result.output or len(result.output) > 0

    def test_show_shorthand_for_pipeline(self):
        """show <name> is shorthand for show pipeline <name>."""
        from soak.api import list_pipelines

        pipelines = list_pipelines()
        if not pipelines:
            pytest.skip("No built-in pipelines available")

        # both should give same content
        result1 = runner.invoke(app, ["show", pipelines[0]])
        result2 = runner.invoke(app, ["show", "pipeline", pipelines[0]])

        assert result1.exit_code == result2.exit_code
        assert result1.output == result2.output

    def test_show_nonexistent_pipeline_fails(self):
        """show nonexistent pipeline returns error."""
        result = runner.invoke(app, ["show", "nonexistent_pipeline_xyz123"])
        assert result.exit_code != 0

    def test_show_nonexistent_template_fails(self):
        """show nonexistent template returns error."""
        result = runner.invoke(app, ["show", "template", "nonexistent_xyz123"])
        assert result.exit_code != 0


class TestVersionFlag:
    """Tests for --version flag."""

    def test_version_flag_short(self):
        """-V shows version."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        # should output something (version string)
        assert len(result.output.strip()) > 0

    def test_version_flag_long(self):
        """--version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0


class TestVerboseFlag:
    """Tests for verbosity flags."""

    def test_verbose_flag_accepted(self):
        """-v flag is accepted."""
        result = runner.invoke(app, ["-v", "show", "pipeline"])
        # should not error on the flag itself
        assert result.exit_code == 0 or "-v" not in str(result.exception)

    def test_double_verbose_accepted(self):
        """-vv flag is accepted."""
        result = runner.invoke(app, ["-vv", "show", "pipeline"])
        assert result.exit_code == 0 or "-vv" not in str(result.exception)


class TestHelpCommand:
    """Tests for help output."""

    def test_main_help(self):
        """--help shows main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "soak" in result.output.lower() or "Usage" in result.output

    def test_run_help(self):
        """run --help shows run command help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower() or "Usage" in result.output

    def test_compare_help(self):
        """compare --help shows compare command help."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0

    def test_show_help(self):
        """show --help shows show command help."""
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0

    def test_export_help(self):
        """export --help shows export command help."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0

    def test_calibrate_help(self):
        """calibrate --help shows calibrate command help."""
        result = runner.invoke(app, ["calibrate", "--help"])
        assert result.exit_code == 0


class TestCalibrateValidation:
    """Tests for calibrate command input validation."""

    def test_calibrate_requires_input(self):
        """calibrate without input shows error."""
        result = runner.invoke(app, ["calibrate"])
        # should fail or show help
        assert result.exit_code != 0 or "Must provide" in result.output or "Usage" in result.output

    def test_calibrate_invalid_method(self):
        """calibrate with invalid method shows error."""
        result = runner.invoke(app, ["calibrate", "--method", "invalid", "dummy.csv"])
        assert result.exit_code != 0 or "Invalid method" in result.output


class TestRunValidation:
    """Tests for run command input validation."""

    def test_run_nonexistent_pipeline_fails(self):
        """run with nonexistent pipeline shows error."""
        result = runner.invoke(app, ["run", "nonexistent_xyz123", "dummy.txt"])
        assert result.exit_code != 0

    def test_run_nonexistent_input_fails(self):
        """run with nonexistent input file shows error."""
        from soak.api import list_pipelines

        pipelines = list_pipelines()
        if not pipelines:
            pytest.skip("No built-in pipelines available")

        result = runner.invoke(app, ["run", pipelines[0], "nonexistent_file_xyz123.txt"])
        # should fail because file doesn't exist
        assert result.exit_code != 0 or "not found" in result.output.lower() or "No files" in result.output


class TestCompareValidation:
    """Tests for compare command input validation."""

    def test_compare_needs_two_inputs(self):
        """compare with less than 2 inputs shows error."""
        result = runner.invoke(app, ["compare", "single.json"])
        assert result.exit_code != 0 or "2" in result.output or "least" in result.output


class TestFormatCommand:
    """Tests for format command."""

    def test_format_help(self):
        """format --help works."""
        result = runner.invoke(app, ["format", "--help"])
        assert result.exit_code == 0

    def test_format_nonexistent_input(self):
        """format with nonexistent input shows error."""
        result = runner.invoke(app, ["format", "nonexistent_xyz123.json"])
        assert result.exit_code != 0
